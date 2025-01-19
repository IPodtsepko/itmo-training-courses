update
    Students
set
    Debts = (
        select
            count(distinct CourseId)
        from
            Students S
            left join Plan using (GroupId)
            left join Marks using (StudentId, CourseId)
        where
            S.StudentId = Students.StudentId
            and Mark is null
    )
where
    GroupId = (
        select
            GroupId
        from
            Groups G
        where
            GroupName = :GroupName
    );
