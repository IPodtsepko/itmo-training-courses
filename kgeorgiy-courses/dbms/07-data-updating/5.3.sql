create view Debts as
select
    StudentId,
    count(distinct CourseId) as Debts
from
    Students
    left join Plan using (GroupId)
    left join Marks using (StudentId, CourseId)
where
    Mark is null
group by
    StudentId
having
    count(CourseId) <> 0;
