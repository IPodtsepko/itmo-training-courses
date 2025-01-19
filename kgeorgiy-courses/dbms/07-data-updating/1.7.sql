delete from
    Students
where
    StudentId in (
        select
            StudentId
        from
            Students
            left join Plan using (GroupId)
            left join Marks using (StudentId, CourseId)
        where
            Mark is null
        group by
            StudentId
        having
            count(distinct CourseId) > 1
    );
